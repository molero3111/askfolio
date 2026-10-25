from app import graph
from app.rate_limit import MessageLimiter


def _setup(monkeypatch):
    sent: list[tuple[int, str]] = []
    monkeypatch.setattr(graph, "_send_telegram", lambda c, text, _m: sent.append((c, text)))
    monkeypatch.setattr(graph, "_message_limiter", MessageLimiter(100, 100, 100))
    monkeypatch.setattr(graph, "_retrieve_context", lambda _t: "ctx")

    def fake_llm(messages):
        if "explode" in messages[-1]["content"]:
            raise AssertionError("not an LLMError")
        return "fine"

    monkeypatch.setattr(graph, "_call_llm", fake_llm)
    return sent


def test_crash_in_one_message_does_not_drop_the_batch(monkeypatch):
    sent = _setup(monkeypatch)

    def bad_retrieve(text):
        if "explode" in text:
            raise RuntimeError("rag down")
        return "ctx"

    monkeypatch.setattr(graph, "_retrieve_context", bad_retrieve)
    state = {
        "offset": None,
        "pending_messages": [(1, 10, "first"), (2, 11, "explode now"), (3, 12, "third")],
        "conversations": {},
    }
    while state["pending_messages"]:
        state = {**state, **graph.process_message_node(state)}
    assert sent == [(1, "fine"), (2, graph.ERROR_REPLY), (3, "fine")]
    assert state["pending_messages"] == []
    assert "1" in state["conversations"] and "3" in state["conversations"]
