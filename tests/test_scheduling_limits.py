from app import graph
from app.conversation import ConversationState, SchedulingMode
from app.rate_limit import MessageLimiter

SLOTS = [
    {"start": "2030-01-01T10:00:00+00:00", "end": "2030-01-01T11:00:00+00:00", "label": "Slot one"},
    {"start": "2030-01-01T12:00:00+00:00", "end": "2030-01-01T13:00:00+00:00", "label": "Slot two"},
]


def _setup(monkeypatch, limiter=None):
    sent: list[str] = []
    calls = {"slots": 0, "tz_llm": 0, "qa_llm": 0}

    def fake_slots(**_kw):
        calls["slots"] += 1
        return list(SLOTS)

    def fake_tz(_text):
        calls["tz_llm"] += 1
        return ""

    monkeypatch.setattr(graph, "_send_telegram", lambda _c, text, _m: sent.append(text))
    monkeypatch.setattr(graph, "get_available_slots", fake_slots)
    monkeypatch.setattr(graph, "is_calendar_configured", lambda: True)
    monkeypatch.setattr(graph, "_resolve_timezone_llm", fake_tz)
    monkeypatch.setattr(graph, "TIMEZONE_ALIAS_FAST_PATH", False)
    monkeypatch.setattr(
        graph, "_message_limiter", limiter or MessageLimiter(100, 100, 100)
    )
    return sent, calls


def test_booking_disabled_never_touches_calendar(monkeypatch):
    sent, calls = _setup(monkeypatch)
    monkeypatch.setattr(graph, "BOOKING_ENABLED", False)
    conv = ConversationState()
    graph._handle_scheduling(conv, "I want to schedule a meeting", 1, 10)
    assert calls["slots"] == 0
    assert sent == [graph.BOOKING_DISABLED_REPLY]
    assert conv.mode == SchedulingMode.IDLE


def test_timezone_request_rate_limited_skips_llm(monkeypatch):
    sent, calls = _setup(monkeypatch, MessageLimiter(0, 100, 100))
    monkeypatch.setattr(graph, "BOOKING_ENABLED", True)
    conv = ConversationState()
    conv.mode = SchedulingMode.AWAITING_SLOT_SELECTION
    conv.proposed_slots = list(SLOTS)
    graph._handle_scheduling(conv, "show in Pacific Time", 1, 10)
    assert calls["tz_llm"] == 0
    assert sent == [graph.RATE_LIMIT_REPLY]
    # a slot number still works without touching the limiter
    graph._handle_scheduling(conv, "2", 1, 11)
    assert conv.mode == SchedulingMode.AWAITING_RECRUITER_INFO


def test_qa_rate_limited_notify_then_silent(monkeypatch):
    sent, _ = _setup(monkeypatch, MessageLimiter(0, 100, 100))
    monkeypatch.setattr(
        graph, "_retrieve_context", lambda _t: (_ for _ in ()).throw(AssertionError("RAG called"))
    )
    state = {
        "offset": None,
        "pending_messages": [(1, 10, "What is your stack?"), (1, 11, "And your education?")],
        "conversations": {},
    }
    out = graph.process_message_node(state)
    assert sent == [graph.RATE_LIMIT_REPLY]
    assert len(out["pending_messages"]) == 1
    out = graph.process_message_node({**state, **out})
    assert sent == [graph.RATE_LIMIT_REPLY]
    assert out["pending_messages"] == []
    assert out["conversations"]["1"]["history"] == []
