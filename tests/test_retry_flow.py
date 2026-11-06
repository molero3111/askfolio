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
