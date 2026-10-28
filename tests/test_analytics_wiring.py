import pytest

from app import graph
from app.llm_client import LLMUnavailable
from app.rate_limit import MessageLimiter


@pytest.fixture
def flow(monkeypatch):
    """Fake Telegram, RAG, LLM and analytics; one shared log records the order of calls."""
    log: list[tuple] = []
    state = {"llm": lambda _m: "An answer."}

    monkeypatch.setattr(graph, "_send_telegram", lambda c, text, m: log.append(("send", c, text)))
    monkeypatch.setattr(graph, "_retrieve_context", lambda _t: "context")
    monkeypatch.setattr(graph, "_call_llm", lambda m: state["llm"](m))
    monkeypatch.setattr(graph, "BOOKING_ENABLED", False)
    monkeypatch.setattr(graph, "_message_limiter", MessageLimiter(100, 100, 100))
    monkeypatch.setattr(
        graph.analytics,
        "track_question",
        lambda chat_id, question, outcome, **kw: log.append(("track", chat_id, question, outcome, kw)),
    )
    monkeypatch.setattr(graph.analytics, "track_first_seen", lambda chat_id: log.append(("first", chat_id)))
    return log, state


def _run(*messages):
    out = {"offset": None, "pending_messages": list(messages), "conversations": {}}
    while out["pending_messages"]:
        out = {**out, **graph.process_message_node(out)}
    return out


def _tracks(log):
    return [e for e in log if e[0] == "track"]


def test_answered_question_is_tracked_after_send(flow):
    log, _ = flow
    _run((7, 1, "What stack?"))

    (track,) = _tracks(log)
    assert track[1:4] == (7, "What stack?", "answered")
    assert isinstance(track[4]["latency_ms"], int) and track[4]["latency_ms"] >= 0
    kinds = [e[0] for e in log]
    assert kinds.index("send") < kinds.index("track")


def test_unavailable_is_tracked(flow):
    log, state = flow

    def down(_m):
        raise LLMUnavailable("all models failed")

    state["llm"] = down
    _run((7, 1, "What stack?"))
    (track,) = _tracks(log)
    assert track[3] == "llm_unavailable"
    assert track[4].get("latency_ms") is None


def test_empty_reply_is_error(flow):
    log, state = flow
    state["llm"] = lambda _m: "<think>only reasoning</think>"
    _run((7, 1, "What stack?"))
    assert [t[3] for t in _tracks(log)] == ["error"]
    assert ("send", 7, graph.ERROR_REPLY) in log


def test_rate_limited_is_tracked_with_scope(flow, monkeypatch):
    log, _ = flow
    monkeypatch.setattr(graph, "_message_limiter", MessageLimiter(1, 100, 100))
    _run((7, 1, "first"), (7, 2, "second"), (7, 3, "third"))

    tracks = _tracks(log)
    assert [t[3] for t in tracks] == ["answered", "rate_limited", "rate_limited"]
    assert tracks[1][2] == "second" and tracks[1][4]["scope"] == "minute"
    assert tracks[2][2] == "third" and tracks[2][4]["scope"] == "minute"
    # The notice goes out before the track; the silent refusal sends nothing
    sends = [e for e in log if e[0] == "send"]
    assert len(sends) == 2
    assert log.index(sends[1]) < log.index(tracks[1])


def test_crash_is_tracked_as_error(flow, monkeypatch):
    log, _ = flow

    def boom(_t):
        raise RuntimeError("db down")

    monkeypatch.setattr(graph, "_retrieve_context", boom)
    out = _run((7, 1, "first"), (8, 2, "second"))

    assert [(t[1], t[2], t[3]) for t in _tracks(log)] == [(7, "first", "error"), (8, "second", "error")]
    assert out["pending_messages"] == []


def test_first_seen_called_per_message(flow):
    log, _ = flow
    _run((7, 1, "a"), (7, 2, "b"), (8, 3, "c"))
    assert [e for e in log if e[0] == "first"] == [("first", 7), ("first", 7), ("first", 8)]


def test_analytics_failure_does_not_change_reply(flow, monkeypatch):
    log, _ = flow

    def broken(*a, **k):
        raise RuntimeError("analytics bug")

    monkeypatch.setattr(graph.analytics, "track_question", broken)
    monkeypatch.setattr(graph.analytics, "track_first_seen", broken)
    _run((7, 1, "a"), (8, 2, "b"))

    assert [e for e in log if e[0] == "send"] == [("send", 7, "An answer."), ("send", 8, "An answer.")]


def test_scheduling_crash_is_not_tracked(flow, monkeypatch):
    # Scheduling replies can hold a recruiter's name; that flow is out of analytics scope
    log, _ = flow
    monkeypatch.setattr(graph, "BOOKING_ENABLED", True)

    def boom(*a, **k):
        raise RuntimeError("calendar bug")

    monkeypatch.setattr(graph, "_handle_scheduling", boom)
    _run((7, 1, "Can we schedule a meeting?"))

    assert ("send", 7, graph.ERROR_REPLY) in log
    assert _tracks(log) == []


def test_first_seen_runs_after_reply(flow):
    log, _ = flow
    _run((7, 1, "What stack?"))
    kinds = [e[0] for e in log]
    assert kinds.index("send") < kinds.index("first")
