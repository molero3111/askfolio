import json
import logging

import pytest

import main
from app import contact_guard, graph
from app.contact_guard import ContactGuard, get_guard
from app.llm_client import LLMUnavailable
from app.rate_limit import MessageLimiter

EMAIL = "jane.doe@example.com"
PHONE = "+1 555 010 1234"
CV = {"personal_info": {"email": EMAIL, "phone": PHONE}}


@pytest.fixture
def flow(monkeypatch):
    sent, tracked = [], []
    llm = {"reply": "An answer."}
    monkeypatch.setattr(graph, "_send_telegram", lambda _c, text, _m: sent.append(text))
    monkeypatch.setattr(graph, "_retrieve_context", lambda _t: "context")
    monkeypatch.setattr(graph, "_call_llm", lambda _m: llm["reply"])
    monkeypatch.setattr(graph, "BOOKING_ENABLED", False)
    monkeypatch.setattr(graph, "_message_limiter", MessageLimiter(100, 100, 100))
    monkeypatch.setattr(graph.analytics, "track_first_seen", lambda _c: None)
    monkeypatch.setattr(
        graph.analytics, "track_question", lambda _c, _q, outcome, **_k: tracked.append(outcome)
    )
    monkeypatch.setattr(contact_guard, "get_guard", lambda: ContactGuard.from_cv(CV))
    return sent, tracked, llm


def _ask() -> list[dict]:
    state = {"offset": None, "pending_messages": [(1, 10, "How do I reach him?")], "conversations": {}}
    return graph.process_message_node(state)["conversations"]["1"]["history"]


@pytest.mark.parametrize(
    "reply",
    [
        f"Write to {EMAIL} today.",
        "Write to jane.doe [at] example dot com today.",
        "Write to jane.doe arroba example punto com today.",
        f"Call {PHONE}.",
        "Call 1-555-010-1234.",
        "Call 555 010 1234.",
        "Call (555) 010/1234.",
        "Call ٥٥٥٠١٠١٢٣٤.",
        r"Write to jane\.doe@example\.com today.",
        r"Write to jane.doe\@example.com today.",
        "Write to jane.doe [ at ] example.com today.",
        "Write to jane.doe {at} example.com today.",
        "Write to jane.doe<at>example<dot>com today.",
        "Write to jane.doe -at- example -dot- com today.",
        "Write to jane.doe [arroba] example [punto] com today.",
        "Write to jane doe at example dot com today.",
        "**Email**jane.doe@example.com",
        "Write to jane.doe+x@example.com today.",
    ],
)
def test_leaking_reply_is_withheld(flow, reply):
    sent, tracked, llm = flow
    llm["reply"] = reply

    history = _ask()

    assert sent == [graph.CONTACT_WITHHELD_REPLY]
    assert history[-1]["content"] == graph.CONTACT_WITHHELD_REPLY
    assert tracked == ["answered"]


def test_clean_reply_is_sent_unchanged(flow):
    sent, _, llm = flow
    llm["reply"] = "He has 6 years of Python experience."

    history = _ask()

    assert sent == ["He has 6 years of Python experience."]
    assert history[-1]["content"] == "He has 6 years of Python experience."


@pytest.mark.parametrize(
    "reply",
    [
        "See github.com/jane.doe and example.com for details.",
        "Jane Doe built example.com in 2024.",
        "jane.doe is his handle at example.org.",
    ],
)
def test_handle_and_domain_apart_are_not_a_leak(reply):
    assert ContactGuard.from_cv(CV).leaks(reply) == []


def test_without_a_guard_the_reply_is_sent_unchanged(flow, monkeypatch):
    sent, _, llm = flow
    llm["reply"] = f"Write to {EMAIL}."
    monkeypatch.setattr(contact_guard, "get_guard", lambda: None)

    _ask()

    assert sent == [f"Write to {EMAIL}."]


def test_log_names_labels_never_values(flow, caplog):
    _, _, llm = flow
    llm["reply"] = f"Write to {EMAIL} or call {PHONE}."

    with caplog.at_level(logging.DEBUG):
        _ask()

    assert "[contact_guard] reply withheld (owner email, owner phone)" in caplog.text
    assert EMAIL not in caplog.text
    assert "555" not in caplog.text


def test_guard_error_gives_the_error_reply_not_the_answer(flow, monkeypatch):
    sent, tracked, llm = flow
    llm["reply"] = f"Write to {EMAIL}."

    def broken():
        raise RuntimeError("boom")

    monkeypatch.setattr(contact_guard, "get_guard", broken)

    _ask()

    assert sent == [graph.ERROR_REPLY]
    assert tracked == ["error"]


def test_error_reply_skips_the_guard(flow, monkeypatch):
    sent, _, _ = flow

    def down(_m):
        raise LLMUnavailable("down")

    def forbidden():
        raise AssertionError("guard consulted")

    monkeypatch.setattr(graph, "_call_llm", down)
    monkeypatch.setattr(contact_guard, "get_guard", forbidden)

    _ask()

    assert sent == [graph.UNAVAILABLE_REPLY]


def test_scheduling_path_skips_the_guard(flow, monkeypatch):
    def forbidden():
        raise AssertionError("guard consulted")

    handled = []
    monkeypatch.setattr(contact_guard, "get_guard", forbidden)
    monkeypatch.setattr(graph, "_routes_to_scheduling", lambda _c, _t: True)
    monkeypatch.setattr(graph, "_handle_scheduling", lambda *a: handled.append(a))

    state = {"offset": None, "pending_messages": [(1, 10, "book a call")], "conversations": {}}
    graph.process_message_node(state)

    assert len(handled) == 1


def _use_file(monkeypatch, path) -> None:
    monkeypatch.setattr(contact_guard.config, "INGEST_KNOWLEDGE_JSON", str(path))
    get_guard.cache_clear()


def test_get_guard_reads_the_knowledge_json(tmp_path, monkeypatch):
    path = tmp_path / "cv.json"
    path.write_text(f'{{"personal_info": {{"email": "{EMAIL}", "phone": "{PHONE}"}}}}')
    _use_file(monkeypatch, path)

    guard = get_guard()

    assert guard is not None
    assert guard.leaks(f"Write to {EMAIL}") == ["owner email"]


@pytest.mark.parametrize("content", [None, "{not json", "[]", '{"personal_info": {}}'])
def test_get_guard_warns_once_and_returns_none(tmp_path, monkeypatch, caplog, content):
    path = tmp_path / "cv.json"
    if content is not None:
        path.write_text(content)
    _use_file(monkeypatch, path)

    with caplog.at_level(logging.WARNING):
        assert get_guard() is None
        assert get_guard() is None

    warnings = [r for r in caplog.records if r.getMessage().startswith("[contact_guard]")]
    assert len(warnings) == 1


@pytest.mark.parametrize("prefix", ["-", ".", "+", "%", "_"])
def test_email_right_after_a_symbol_is_caught(prefix):
    guard = ContactGuard.from_cv(CV)
    assert guard.leaks(f"Write to {prefix}{EMAIL} today.") == ["owner email"]


@pytest.mark.parametrize("phone", ["123", "12 34 56", "n/a"])
def test_too_short_a_phone_counts_as_absent(phone):
    guard = ContactGuard.from_cv({"personal_info": {"email": EMAIL, "phone": phone}})
    assert guard.phone_digits is None
    assert guard.leaks("He shipped 3 projects in 2012 and 123456 lines.") == []


@pytest.mark.parametrize(
    "info,missing",
    [({"email": EMAIL}, "owner phone"), ({"phone": PHONE}, "owner email")],
)
def test_get_guard_warns_about_a_missing_half(tmp_path, monkeypatch, caplog, info, missing):
    path = tmp_path / "cv.json"
    path.write_text(json.dumps({"personal_info": info}))
    _use_file(monkeypatch, path)

    with caplog.at_level(logging.WARNING):
        assert get_guard() is not None

    assert f"[contact_guard] {missing} not in the knowledge JSON" in caplog.text
    assert "jane.doe" not in caplog.text
    assert "555" not in caplog.text


def test_get_guard_treats_a_malformed_email_as_missing(tmp_path, monkeypatch, caplog):
    path = tmp_path / "cv.json"
    path.write_text(json.dumps({"personal_info": {"email": "jane.doe@", "phone": PHONE}}))
    _use_file(monkeypatch, path)

    with caplog.at_level(logging.WARNING):
        guard = get_guard()

    assert guard is not None
    assert guard.email is None
    assert "[contact_guard] owner email not in the knowledge JSON" in caplog.text


def test_startup_loads_the_guard_and_reports_a_problem(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(main, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(main, "LLM_MODEL", "a/b")
    monkeypatch.setattr(main, "get_vectorstore", lambda: None)
    _use_file(monkeypatch, tmp_path / "missing.json")

    with caplog.at_level(logging.ERROR):
        main.startup()

    assert "[contact_guard] knowledge JSON unusable" in caplog.text
