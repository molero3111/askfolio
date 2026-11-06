import logging

import pytest

from app import graph, retry_queue
from app.llm_client import LLMCreditExhausted, LLMTemporary, LLMUnavailable
from app.rate_limit import MessageLimiter
from app.retry_queue import RetryQueue

MARKER = "fabricated private phrase"
CHAT = 1
MSG = 10


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class Flow:
    """Fake Telegram, RAG, LLM and analytics around the real queue, with a fake clock."""

    def __init__(self, tmp_path):
        self.clock = Clock()
        self.path = tmp_path / "flow_queue.json"
        self.queue = self.new_queue()
        self.sends: list[tuple] = []
        self.send_status = "sent"
        self.llm = lambda _messages: "An answer."
        self.llm_calls = 0
        self.tracks: list[tuple] = []

    def new_queue(self, **overrides) -> RetryQueue:
        settings = {"interval": 1800, "after_success": 60, "max_age": 86400, "max_items": 20, "max_per_chat": 3}
        return RetryQueue(self.path, clock=self.clock, **{**settings, **overrides})


def _raises(exc):
    def llm(_messages):
        raise exc

    return llm


@pytest.fixture
def flow(tmp_path, monkeypatch):
    f = Flow(tmp_path)

    def send(chat_id, text, message_id):
        f.sends.append((chat_id, text, message_id))
        return f.send_status

    def call_llm(messages):
        f.llm_calls += 1
        return f.llm(messages)

    monkeypatch.setattr(graph, "_send_telegram", send)
    monkeypatch.setattr(graph, "_retrieve_context", lambda _t: "context")
    monkeypatch.setattr(graph, "_call_llm", call_llm)
    monkeypatch.setattr(graph, "BOOKING_ENABLED", False)
    monkeypatch.setattr(graph, "_message_limiter", MessageLimiter(100, 100, 100))
    monkeypatch.setattr(retry_queue, "get_queue", lambda: f.queue)
    monkeypatch.setattr(
        graph.analytics,
        "track_question",
        lambda chat_id, question, outcome, **kw: f.tracks.append((outcome, kw)),
    )
    monkeypatch.setattr(graph.analytics, "track_first_seen", lambda chat_id: None)
    return f


def _ask(text="What does he do?", conversations=None):
    state = {"offset": None, "pending_messages": [(CHAT, MSG, text)], "conversations": conversations or {}}
    return graph.process_message_node(state)["conversations"]


def _history(conversations, chat=CHAT):
    return conversations.get(str(chat), {}).get("history", [])


def test_temporary_failure_is_queued_with_notice(flow):
    flow.llm = _raises(LLMTemporary("LLM unavailable (HTTP 429)"))

    conversations = _ask()

    assert flow.sends == [(CHAT, graph.QUEUED_REPLY, MSG)]
    assert len(flow.queue) == 1
    assert flow.tracks == [("queued", {"latency_ms": None})]
    assert _history(conversations) == []


def test_same_question_again_gets_the_notice_without_a_new_entry(flow):
    flow.llm = _raises(LLMTemporary("down"))

    _ask("Same question")
    _ask("same question ")

    assert [s[1] for s in flow.sends] == [graph.QUEUED_REPLY, graph.QUEUED_REPLY]
    assert len(flow.queue) == 1


@pytest.mark.parametrize("exc", [LLMUnavailable("rejected (HTTP 400)"), LLMCreditExhausted("credit limit reached")])
def test_permanent_failure_is_not_queued(flow, exc):
    flow.llm = _raises(exc)

    _ask()

    assert flow.sends == [(CHAT, graph.UNAVAILABLE_REPLY, MSG)]
    assert len(flow.queue) == 0
    assert flow.tracks[0][0] == "llm_unavailable"


def test_queue_full_gets_plain_unavailable(flow):
    flow.queue = flow.new_queue(max_items=0)
    flow.llm = _raises(LLMTemporary("down"))

    _ask()

    assert flow.sends == [(CHAT, graph.UNAVAILABLE_REPLY, MSG)]
    assert flow.tracks[0][0] == "llm_unavailable"


def test_queue_write_failure_gets_plain_unavailable(flow, monkeypatch):
    def boom(*a):
        raise OSError("read-only")

    monkeypatch.setattr(retry_queue, "atomic_write_secret", boom)
    flow.llm = _raises(LLMTemporary("down"))

    _ask()

    assert flow.sends == [(CHAT, graph.UNAVAILABLE_REPLY, MSG)]
    assert len(flow.queue) == 0
    assert flow.tracks[0][0] == "llm_unavailable"


def test_failed_answer_send_is_queued_without_notice(flow):
    flow.send_status = "retry"

    conversations = _ask()

    assert flow.sends == [(CHAT, "An answer.", MSG)]
    assert len(flow.queue) == 1
    assert flow.tracks == [("queued", {"latency_ms": None})]
    assert _history(conversations) == []


def test_failed_answer_send_with_full_queue_is_undeliverable(flow):
    flow.queue = flow.new_queue(max_items=0)
    flow.send_status = "retry"

    conversations = _ask()

    assert flow.tracks[0][0] == "undeliverable"
    assert _history(conversations) == []


def test_gone_answer_send_is_not_queued(flow):
    flow.send_status = "gone"

    conversations = _ask()

    assert len(flow.queue) == 0
    assert flow.tracks == [("undeliverable", {"latency_ms": None})]
    assert _history(conversations) == []


def test_answered_flow_is_unchanged(flow):
    conversations = _ask("Hello there")

    assert flow.sends == [(CHAT, "An answer.", MSG)]
    assert len(flow.queue) == 0
    assert flow.tracks[0][0] == "answered"
    assert flow.tracks[0][1]["latency_ms"] is not None
    assert [m["content"] for m in _history(conversations)] == ["Hello there", "An answer."]


def test_fake_send_returning_none_counts_as_sent(flow, monkeypatch):
    monkeypatch.setattr(graph, "_send_telegram", lambda *a: None)

    _ask()

    assert len(flow.queue) == 0
    assert flow.tracks[0][0] == "answered"


def test_scheduling_is_never_queued(flow, monkeypatch):
    monkeypatch.setattr(graph, "BOOKING_ENABLED", True)
    monkeypatch.setattr(graph, "_handle_scheduling", lambda *a: None)
    flow.send_status = "retry"
    flow.llm = _raises(LLMTemporary("down"))

    _ask("Can we schedule a meeting?")

    assert len(flow.queue) == 0


def test_queue_contents_are_never_logged(flow, caplog):
    flow.llm = _raises(LLMTemporary("down"))

    with caplog.at_level(logging.INFO):
        _ask(MARKER)

    assert len(flow.queue) == 1
    assert MARKER not in caplog.text


# The worker


def _due(flow, text="What does he do?", chat=CHAT, msg=MSG):
    """Queue a question and move the clock to its first attempt."""
    flow.queue.add(chat, msg, text)
    flow.clock.now += 1800


@pytest.fixture
def retries(flow, monkeypatch):
    log: list[tuple] = []
    monkeypatch.setattr(
        graph.analytics,
        "track_retry",
        lambda chat_id, outcome, **kw: log.append((chat_id, outcome, kw)),
    )
    return log


def test_late_answer_replies_to_original_message(flow, retries):
    _due(flow)

    graph.process_retry_queue({})

    assert flow.sends == [(CHAT, graph.LATE_REPLY_PREFIX + "An answer.", MSG)]
    assert len(flow.queue) == 0
    assert retries == [(CHAT, "delivered_late", {"attempts": 0, "waited_s": 1800})]


def test_late_delivery_is_not_counted_as_a_new_question(flow, retries):
    _due(flow)

    graph.process_retry_queue({})

    assert flow.tracks == []


def test_nothing_runs_before_due(flow, retries):
    flow.queue.add(CHAT, MSG, "q")
    flow.clock.now += 1799

    graph.process_retry_queue({})

    assert flow.llm_calls == 0
    assert flow.sends == []


def test_one_attempt_per_call(flow, retries):
    flow.queue.add(CHAT, MSG, "first")
    flow.queue.add(2, 20, "second")
    flow.clock.now += 1800

    graph.process_retry_queue({})
    graph.process_retry_queue({})

    assert flow.llm_calls == 1
    flow.clock.now += 60
    graph.process_retry_queue({})
    assert flow.llm_calls == 2
    assert len(flow.queue) == 0


def test_still_down_keeps_item(flow, retries):
    _due(flow)
    flow.llm = _raises(LLMTemporary("down"))

    graph.process_retry_queue({})

    assert flow.sends == []
    assert retries == []
    assert len(flow.queue) == 1
    flow.clock.now += 1800
    assert flow.queue.next_due().attempts == 1


def test_empty_reply_keeps_item(flow, retries):
    _due(flow)
    flow.llm = lambda _m: "<think>only reasoning</think>"

    graph.process_retry_queue({})

    assert flow.sends == []
    assert len(flow.queue) == 1


def test_late_answer_goes_through_contact_check(flow, retries, monkeypatch):
    email = "jane.doe@example.com"
    guard = graph.contact_guard.ContactGuard.from_cv({"personal_info": {"email": email}})
    monkeypatch.setattr(graph.contact_guard, "get_guard", lambda: guard)
    flow.llm = lambda _m: f"Write to {email}"
    _due(flow)
    conversations: dict = {}

    graph.process_retry_queue(conversations)

    assert flow.sends == [(CHAT, graph.LATE_REPLY_PREFIX + graph.CONTACT_WITHHELD_REPLY, MSG)]
    assert email not in str(conversations)


def test_history_holds_answer_without_prefix(flow, retries):
    _due(flow, "My question")
    conversations: dict = {}

    graph.process_retry_queue(conversations)

    assert [m["content"] for m in _history(conversations)] == ["My question", "An answer."]


def test_retry_uses_the_history_the_chat_has_now(flow, retries):
    conversations = _ask("Earlier question")
    seen = []
    flow.llm = lambda messages: seen.append(str(messages)) or "An answer."
    _due(flow, "Later question")

    graph.process_retry_queue(conversations)

    assert "Earlier question" in seen[0]
    assert [m["content"] for m in _history(conversations)][-2:] == ["Later question", "An answer."]


def test_expired_item_gets_apology_once(flow, retries):
    flow.queue.add(CHAT, MSG, "q")
    flow.clock.now += 86400

    graph.process_retry_queue({})
    graph.process_retry_queue({})

    assert flow.sends == [(CHAT, graph.EXPIRED_REPLY, MSG)]
    assert flow.llm_calls == 0
    assert retries == [(CHAT, "expired", {"attempts": 0, "waited_s": 86400})]
    assert not flow.path.exists()


def _expired_item(flow):
    flow.queue.add(CHAT, MSG, "q")
    flow.clock.now += 86400


def test_apology_is_kept_when_sending_may_work_later(flow, retries):
    _expired_item(flow)
    flow.send_status = "retry"

    graph.process_retry_queue({})

    assert len(flow.queue) == 1
    assert retries == []
    assert flow.llm_calls == 0

    flow.send_status = "sent"
    graph.process_retry_queue({})

    assert [s[1] for s in flow.sends] == [graph.EXPIRED_REPLY, graph.EXPIRED_REPLY]
    assert len(flow.queue) == 0
    assert retries[0][1] == "expired"


def test_apology_stops_after_the_first_failed_send(flow, retries):
    flow.queue.add(CHAT, MSG, "a")
    flow.queue.add(2, 20, "b")
    flow.clock.now += 86400
    flow.send_status = "retry"

    graph.process_retry_queue({})

    assert len(flow.sends) == 1
    assert len(flow.queue) == 2


def test_apology_gives_up_after_the_grace_period(flow, retries, caplog):
    _expired_item(flow)
    flow.send_status = "retry"
    graph.process_retry_queue({})
    flow.clock.now += graph.EXPIRED_APOLOGY_GRACE_SECONDS

    with caplog.at_level(logging.WARNING):
        graph.process_retry_queue({})

    assert len(flow.sends) == 1
    assert len(flow.queue) == 0
    assert retries[0][1] == "expired"
    assert "apology not sent" in caplog.text


def test_blocked_chat_expiry_is_removed(flow, retries):
    _expired_item(flow)
    flow.send_status = "gone"

    graph.process_retry_queue({})

    assert len(flow.queue) == 0
    assert retries[0][1] == "expired"


def test_each_apology_touches_the_heartbeat(flow, retries, monkeypatch):
    touches = []
    monkeypatch.setattr(graph.heartbeat, "touch", lambda: touches.append(1))
    flow.queue.add(CHAT, MSG, "a")
    flow.queue.add(2, 20, "b")
    flow.clock.now += 86400

    graph.process_retry_queue({})

    assert len(touches) == 2


def test_restart_resumes(flow, retries):
    _due(flow, "My question")
    flow.queue = flow.new_queue()

    conversations: dict = {}
    graph.process_retry_queue(conversations)

    assert flow.sends == [(CHAT, graph.LATE_REPLY_PREFIX + "An answer.", MSG)]
    assert len(flow.queue) == 0
    assert not flow.path.exists()
    assert len(_history(conversations)) == 2


def test_blocked_chat_is_dropped(flow, retries):
    _due(flow)
    flow.send_status = "gone"
    conversations: dict = {}

    graph.process_retry_queue(conversations)

    assert len(flow.queue) == 0
    assert retries == [(CHAT, "undeliverable", {"attempts": 0, "waited_s": 1800})]
    assert _history(conversations) == []


def test_late_send_failure_retries(flow, retries):
    _due(flow)
    flow.send_status = "retry"
    conversations: dict = {}

    graph.process_retry_queue(conversations)

    assert len(flow.queue) == 1
    assert retries == []
    assert _history(conversations) == []


def test_attempt_touches_heartbeat_first(flow, retries, monkeypatch):
    order = []
    monkeypatch.setattr(graph.heartbeat, "touch", lambda: order.append("touch"))
    flow.llm = lambda _m: order.append("llm") or "An answer."
    _due(flow)

    graph.process_retry_queue({})

    assert order == ["touch", "llm"]


def test_worker_never_raises_and_logs_no_text(flow, retries, monkeypatch, caplog):
    def boom():
        raise RuntimeError(MARKER)

    monkeypatch.setattr(retry_queue, "get_queue", boom)
    with caplog.at_level(logging.INFO):
        graph.process_retry_queue({})
    assert "RuntimeError" in caplog.text
    assert MARKER not in caplog.text


def test_failed_attempt_logs_no_text(flow, retries, caplog):
    _due(flow, MARKER)
    flow.llm = _raises(RuntimeError(MARKER))

    with caplog.at_level(logging.INFO):
        graph.process_retry_queue({})

    assert "RuntimeError" in caplog.text
    assert MARKER not in caplog.text


def test_worker_lets_shutdown_through(flow, retries):
    _due(flow)
    flow.llm = _raises(SystemExit(0))

    with pytest.raises(SystemExit):
        graph.process_retry_queue({})

    assert len(flow.queue) == 1
    assert flow.path.exists()


# Asking again


def test_asking_again_after_recovery_answers_once(flow, retries):
    flow.llm = _raises(LLMTemporary("down"))
    conversations = _ask("My question")
    flow.llm = lambda _m: "An answer."

    conversations = _ask("my question ", conversations)
    flow.clock.now += 1800
    graph.process_retry_queue(conversations)

    assert [s[1] for s in flow.sends] == [graph.QUEUED_REPLY, "An answer."]
    assert len(flow.queue) == 0
    assert not flow.path.exists()
    assert retries == []
    assert [m["content"] for m in _history(conversations)] == ["my question ", "An answer."]


def test_answering_another_question_keeps_the_queued_one(flow):
    flow.llm = _raises(LLMTemporary("down"))
    _ask("First question")
    flow.llm = lambda _m: "An answer."

    _ask("Second question")

    assert len(flow.queue) == 1


def test_same_question_from_another_chat_keeps_the_queued_one(flow):
    flow.queue.add(2, 20, "My question")

    _ask("My question")

    assert len(flow.queue) == 1


def test_failed_resend_keeps_the_queued_one(flow):
    flow.llm = _raises(LLMTemporary("down"))
    _ask("My question")
    flow.llm = lambda _m: "An answer."
    flow.send_status = "retry"

    _ask("My question")

    assert len(flow.queue) == 1
