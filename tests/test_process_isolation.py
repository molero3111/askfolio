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


def test_reasoning_only_reply_becomes_error_reply(monkeypatch):
    sent = _setup(monkeypatch)
    monkeypatch.setattr(graph, "_call_llm", lambda _m: "<think>only thoughts</think>")
    state = {"offset": None, "pending_messages": [(1, 10, "hello")], "conversations": {}}
    out = graph.process_message_node(state)
    assert sent == [(1, graph.ERROR_REPLY)]
    assert "No reply could be extracted" not in str(out["conversations"])


def test_message_text_is_not_logged(monkeypatch, caplog):
    import logging

    _setup(monkeypatch)
    secret_text = "my private phrase about salary"
    state = {"offset": None, "pending_messages": [(1, 10, secret_text)], "conversations": {}}
    with caplog.at_level(logging.INFO):
        graph.process_message_node(state)
    assert "private phrase" not in caplog.text
    assert "chat_id=1" in caplog.text


def test_llm_unavailable_gets_unavailable_reply(monkeypatch):
    from app.llm_client import LLMUnavailable

    sent = _setup(monkeypatch)

    def down(_m):
        raise LLMUnavailable("LLM unavailable (HTTP 429)")

    monkeypatch.setattr(graph, "_call_llm", down)
    state = {"offset": None, "pending_messages": [(1, 10, "hello")], "conversations": {}}
    graph.process_message_node(state)
    assert sent == [(1, graph.UNAVAILABLE_REPLY)]


def test_timezone_resolver_does_not_log_raw_text(monkeypatch, caplog):
    import logging

    from app.llm_client import LLMError

    secret = "zz-private-tz-text"
    for behaviour in ("garbage", "llm_error", "crash"):
        def fake(_m, behaviour=behaviour):
            if behaviour == "garbage":
                return "not a zone"
            if behaviour == "llm_error":
                raise LLMError("x")
            raise RuntimeError("boom")

        monkeypatch.setattr(graph, "_call_llm", fake)
        with caplog.at_level(logging.INFO):
            assert graph._resolve_timezone_llm(secret) is None
    assert secret not in caplog.text
    assert f"len={len(secret)}" in caplog.text


def test_langsmith_exception_propagates_from_message_path(monkeypatch):
    import pytest

    _setup(monkeypatch)

    def boom(_t):
        raise RuntimeError("langsmith: failed to post run to api.smith.langchain.com")

    monkeypatch.setattr(graph, "_retrieve_context", boom)
    state = {"offset": None, "pending_messages": [(1, 10, "hi")], "conversations": {}}
    with pytest.raises(RuntimeError):
        graph.process_message_node(state)


def test_normal_exception_is_isolated_not_raised(monkeypatch):
    sent = _setup(monkeypatch)

    def boom(_t):
        raise RuntimeError("rag down")

    monkeypatch.setattr(graph, "_retrieve_context", boom)
    state = {"offset": None, "pending_messages": [(1, 10, "hi")], "conversations": {}}
    out = graph.process_message_node(state)
    assert out["pending_messages"] == []
    assert sent == [(1, graph.ERROR_REPLY)]
