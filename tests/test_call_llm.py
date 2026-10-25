from app import graph
from app.llm_client import LLMResult


def test_call_llm_sends_messages_unchanged(monkeypatch):
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]
    seen = {}

    def fake_chat(msgs):
        seen["messages"] = msgs
        return LLMResult("hi", "m", 1, 1, None, 0.1)

    monkeypatch.setattr(graph, "chat", fake_chat)
    assert graph._call_llm(messages) == "hi"
    assert seen["messages"] == messages
