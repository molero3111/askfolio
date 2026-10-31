import pytest

from app import analytics, graph
from app.llm_client import LLMUnavailable


def test_answer_question_runs_the_qa_pipeline(monkeypatch):
    monkeypatch.setattr(graph, "_retrieve_context", lambda _q: "CTX")
    seen = []

    def fake_llm(messages):
        seen.append(messages)
        return "<think>x</think>**Laravel** expert"

    reply = graph.answer_question(
        "What stack?", [{"role": "user", "content": "hi"}], call_llm=fake_llm
    )

    assert reply == "Laravel expert"
    user_message = seen[0][-1]["content"]
    assert "CTX" in user_message
    assert "Recruiter: hi" in user_message


def test_answer_question_defaults_to_call_llm(monkeypatch):
    monkeypatch.setattr(graph, "_retrieve_context", lambda _q: "CTX")
    monkeypatch.setattr(graph, "_call_llm", lambda _m: "ok")
    assert graph.answer_question("q", []) == "ok"


def test_answer_question_sends_nothing(monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("side effect")

    monkeypatch.setattr(graph, "_retrieve_context", lambda _q: "CTX")
    monkeypatch.setattr(graph, "_send_telegram", forbidden)
    monkeypatch.setattr(analytics, "track_question", forbidden)
    assert graph.answer_question("q", [], call_llm=lambda _m: "fine") == "fine"


def test_answer_question_propagates_llm_errors(monkeypatch):
    monkeypatch.setattr(graph, "_retrieve_context", lambda _q: "CTX")

    def down(_m):
        raise LLMUnavailable("down")

    with pytest.raises(LLMUnavailable):
        graph.answer_question("q", [], call_llm=down)
