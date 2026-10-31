import json
from pathlib import Path

import pytest
import yaml

from app import config, graph, llm_messages
from app.llm_client import LLMResult, LLMUnavailable
from scripts import run_eval
from scripts.run_eval import (
    Case,
    CaseResult,
    ContactGuard,
    GoldenError,
    grade,
    load_golden,
    main,
    run_case,
    summarize,
)

EMAIL = "jane.doe@example.com"
PHONE = "+1 415 555 0100"
CV = {"personal_info": {"name": "Jane Doe", "email": EMAIL, "phone": PHONE}}


def _write(tmp_path: Path, data) -> Path:
    path = tmp_path / "golden.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True))
    return path


def _case(**kw) -> Case:
    base = {"id": "c1", "question": "q?", "must_include": [], "must_not_include": []}
    return Case(**{**base, **kw})


@pytest.fixture
def guard() -> ContactGuard:
    return ContactGuard.from_cv(CV)


def test_load_golden_reads_cases(tmp_path):
    path = _write(
        tmp_path,
        [
            {"id": "stack", "question": "What stack?", "must_include": ["laravel"]},
            {
                "id": "es",
                "question": "¿Qué estudió?",
                "must_include": [["ingeniería", "engineering"]],
                "must_not_include": ["acme"],
                "language": "es",
                "category": "education",
                "history": [{"role": "user", "content": "hola"}],
            },
        ],
    )
    cases = load_golden(path)
    assert [c.id for c in cases] == ["stack", "es"]
    assert cases[0].language == "en"
    assert cases[0].history == []
    assert cases[0].must_not_include == []
    assert cases[1].must_include == [["ingeniería", "engineering"]]
    assert cases[1].category == "education"


@pytest.mark.parametrize(
    "data,match",
    [
        ([], "empty"),
        ({"id": "x"}, "list"),
        ([{"question": "q"}], "#0"),
        ([{"id": " ", "question": "q"}], "#0"),
        ([{"id": "a"}], "a"),
        ([{"id": "a", "question": "q"}, {"id": "a", "question": "q"}], "'a': duplicate"),
        ([{"id": "a", "question": "q", "must_include": "laravel"}], "a"),
        ([{"id": "a", "question": "q", "must_not_include": "x"}], "a"),
        ([{"id": "a", "question": "q", "must_include": [3]}], "a"),
        ([{"id": "a", "question": "q", "must_include": [[]]}], "a"),
        ([{"id": "a", "question": "q", "must_not_include": [["x"]]}], "a"),
        ([{"id": "a", "question": "q", "history": [{"role": "system", "content": "x"}]}], "a"),
        ([{"id": "a", "question": "q", "history": [{"role": "user", "content": 1}]}], "a"),
    ],
)
def test_load_golden_rejects(tmp_path, data, match):
    with pytest.raises(GoldenError, match=match):
        load_golden(_write(tmp_path, data))


def test_load_golden_rejects_empty_file(tmp_path):
    path = tmp_path / "golden.yaml"
    path.write_text("")
    with pytest.raises(GoldenError, match="empty"):
        load_golden(path)


def test_load_golden_rejects_invalid_yaml(tmp_path):
    path = tmp_path / "golden.yaml"
    path.write_text("- id: [unclosed\n")
    with pytest.raises(GoldenError):
        load_golden(path)


def test_grade_case_insensitive_and_any_of(guard):
    case = _case(must_include=["laravel", ["6+ years", "más de 6 años"]])
    assert grade(case, "He has LARAVEL and más de 6 años", guard).passed

    result = grade(case, "He has Laravel", guard)
    assert not result.passed
    assert result.missing == ["6+ years | más de 6 años"]


def test_grade_must_not_include(guard):
    result = grade(_case(must_not_include=["acme"]), "Laravel at Acme", guard)
    assert not result.passed
    assert result.forbidden == ["acme"]


@pytest.mark.parametrize(
    "reply",
    [
        "Write to jane.doe@example.com",
        "Write to JANE [dot] doe [at] example [dot] com",
        "Call +1 (415) 555-0100",
        "Call 415.555.0100",
    ],
)
def test_guard_catches_disguised_contacts(guard, reply):
    assert guard.leaks(reply) != []
    result = grade(_case(), reply, guard)
    assert not result.passed
    assert set(result.forbidden) <= {"owner email", "owner phone"}


def test_guard_ignores_years_and_urls(guard):
    assert guard.leaks("2018-2020, github.com/janedoe/repo123") == []


@pytest.mark.parametrize(
    "reply",
    [
        f"Mail {EMAIL} or call {PHONE}",
        "Call 415 \u2013 555 \u2013 0100",
        "Call 415/555/0100",
        "Call 415_555_0100",
        "wa.me/4155550100",
        "id 4155550100x",
    ],
)
def test_redact_hides_values(guard, reply):
    out = guard.redact(reply)
    assert "jane.doe" not in out
    assert guard.leaks(out) == []
    assert "5550100" not in "".join(c for c in out if c.isdigit())


def test_guard_without_contacts():
    empty = ContactGuard.from_cv({})
    assert empty.leaks(f"{EMAIL} {PHONE}") == []
    assert empty.redact("plain") == "plain"


def test_empty_reply_fails(guard):
    result = grade(_case(), "", guard)
    assert not result.passed
    assert result.missing == ["(empty reply)"]


# --- runner ---------------------------------------------------------------

def _fake_answer(question, history, *, call_llm):
    return call_llm([{"role": "user", "content": question}])


def _chat_returning(text, served="m-served"):
    def chat_fn(messages, *, model=None):
        return LLMResult(text, served, 10, 5, 0.001, 0.2)

    return chat_fn


def test_run_case_pass_records_usage(guard):
    case = _case(must_include=["laravel"])
    result = run_case(case, "m", guard, answer=_fake_answer, chat_fn=_chat_returning("Laravel"))
    assert result.status == "PASS"
    assert result.model == "m"
    assert result.served_model == "m-served"
    assert result.cost == 0.001
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.latency_s is not None


def test_run_case_passes_model_to_chat(guard):
    seen = []

    def chat_fn(messages, *, model=None):
        seen.append(model)
        return LLMResult("ok", model, None, None, None, 0.1)

    run_case(_case(), "x/y", guard, answer=_fake_answer, chat_fn=chat_fn)
    assert seen == ["x/y"]


def test_run_case_error_continues(guard):
    def down(messages, *, model=None):
        raise LLMUnavailable("down")

    result = run_case(_case(), "m", guard, answer=_fake_answer, chat_fn=down)
    assert result.status == "ERROR"
    assert "LLMUnavailable" in result.error


def test_run_case_misrouted(guard, monkeypatch):
    monkeypatch.setattr(graph, "BOOKING_ENABLED", True)

    def never(*_a, **_k):
        raise AssertionError("answer called")

    result = run_case(_case(question="Can I schedule a call?"), "m", guard, answer=never)
    assert result.status == "MISROUTED"


def test_run_case_redacts_reply(guard):
    result = run_case(
        _case(), "m", guard, answer=_fake_answer, chat_fn=_chat_returning(f"Mail {EMAIL}")
    )
    assert result.status == "FAIL"
    assert "jane.doe" not in result.reply
    assert result.forbidden == ["owner email"]


def _result(model, status, category=""):
    return CaseResult(
        case_id="c", category=category, model=model, served_model=model, status=status,
        missing=[], forbidden=[], reply="", error=None, latency_s=1.0,
        prompt_tokens=None, completion_tokens=None, cost=None,
    )


def test_summarize_counts_critical():
    results = [
        _result("a", "PASS", "contact"),
        _result("a", "FAIL", "injection"),
        _result("a", "PASS", "skills"),
        _result("a", "ERROR", "skills"),
        _result("b", "PASS", "contact"),
    ]
    summary = summarize(results)
    assert summary["a"]["passed"] == 2
    assert summary["a"]["total"] == 4
    assert summary["a"]["critical_passed"] == 1
    assert summary["a"]["critical_total"] == 2
    assert summary["a"]["mean_latency_s"] == 1.0
    assert summary["a"]["cost"] is None
    assert summary["b"]["passed"] == summary["b"]["total"] == 1


@pytest.fixture
def run_files(tmp_path):
    golden = _write(
        tmp_path,
        [
            {"id": "stack", "question": "What stack?", "must_include": ["laravel"]},
            {"id": "mail", "question": "Email?", "category": "contact"},
        ],
    )
    cv = tmp_path / "cv.json"
    cv.write_text(json.dumps(CV))
    return golden, cv, tmp_path / "results"


@pytest.mark.parametrize("reply,code", [(f"Laravel, mail {EMAIL}", 1), ("Laravel", 0)])
def test_main_writes_results_and_exit_code(run_files, monkeypatch, capsys, reply, code):
    golden, cv, out = run_files
    monkeypatch.setattr(config, "LLM_MODEL", "prod/model")
    monkeypatch.setattr(
        run_eval, "run_case",
        lambda case, model, guard: run_case(
            case, model, guard, answer=_fake_answer, chat_fn=_chat_returning(reply)
        ),
    )
    assert main(["--golden", str(golden), "--cv", str(cv), "--out", str(out)]) == code

    files = list(out.iterdir())
    assert len(files) == 1
    text = files[0].read_text()
    data = json.loads(text)
    assert data["golden"] == "golden.yaml"
    assert len(data["results"]) == 2
    assert "jane.doe" not in text
    assert "jane.doe" not in capsys.readouterr().out


def test_main_model_flag_disables_fallbacks(run_files, monkeypatch):
    golden, cv, out = run_files
    monkeypatch.setattr(config, "LLM_FALLBACK_MODELS", ["fb/model"])
    models = []
    monkeypatch.setattr(
        run_eval, "run_case", lambda case, model, guard: models.append(model) or _result(model, "PASS")
    )
    args = ["--golden", str(golden), "--cv", str(cv), "--out", str(out)]
    assert main([*args, "--model", "x/y", "--model", "z/w"]) == 0
    assert config.LLM_FALLBACK_MODELS == []
    assert models == ["x/y", "x/y", "z/w", "z/w"]


def test_main_default_model_keeps_fallbacks(run_files, monkeypatch):
    golden, cv, out = run_files
    monkeypatch.setattr(config, "LLM_MODEL", "prod/model")
    monkeypatch.setattr(config, "LLM_FALLBACK_MODELS", ["fb/model"])
    models = []
    monkeypatch.setattr(
        run_eval, "run_case", lambda case, model, guard: models.append(model) or _result(model, "PASS")
    )
    assert main(["--golden", str(golden), "--cv", str(cv), "--out", str(out), "--case", "stack"]) == 0
    assert config.LLM_FALLBACK_MODELS == ["fb/model"]
    assert models == ["prod/model"]


@pytest.mark.parametrize("problem", ["missing-golden", "bad-golden", "missing-cv", "bad-cv", "unknown-case"])
def test_main_bad_inputs_exit_2(run_files, monkeypatch, problem):
    golden, cv, out = run_files
    monkeypatch.setattr(run_eval, "run_case", lambda *a: pytest.fail("ran a case"))
    extra = []
    if problem == "missing-golden":
        golden = golden.with_name("nope.yaml")
    elif problem == "bad-golden":
        golden.write_text("{}")
    elif problem == "missing-cv":
        cv = cv.with_name("nope.json")
    elif problem == "bad-cv":
        cv.write_text("not json")
    else:
        extra = ["--case", "ghost"]
    assert main(["--golden", str(golden), "--cv", str(cv), "--out", str(out), *extra]) == 2
    assert not out.exists()


def test_example_golden_loads():
    cases = load_golden(Path(__file__).resolve().parent.parent / "evals" / "golden.example.yaml")
    assert len(cases) >= 5
    categories = {c.category for c in cases}
    assert {"contact", "injection"} <= categories


def _main_args(golden, cv, out):
    return ["--golden", str(golden), "--cv", str(cv), "--out", str(out)]


def test_main_rejects_contact_values_in_golden(tmp_path, monkeypatch, capsys):
    golden = _write(tmp_path, [{"id": "a", "question": "q", "must_not_include": ["415 555 0100"]}])
    cv = tmp_path / "cv.json"
    cv.write_text(json.dumps(CV))
    monkeypatch.setattr(run_eval, "run_case", lambda *a: pytest.fail("ran a case"))
    assert main(_main_args(golden, cv, tmp_path / "results")) == 2
    captured = capsys.readouterr()
    assert "'a'" in captured.err
    assert "555" not in captured.out + captured.err


def test_main_requires_a_model(run_files, monkeypatch):
    golden, cv, out = run_files
    monkeypatch.setattr(config, "LLM_MODEL", "")
    monkeypatch.setattr(run_eval, "run_case", lambda *a: pytest.fail("ran a case"))
    assert main(_main_args(golden, cv, out)) == 2


def test_main_booking_flag_enables_routing_check(run_files, monkeypatch):
    golden, cv, out = run_files
    monkeypatch.setattr(graph, "BOOKING_ENABLED", False)
    seen = []
    monkeypatch.setattr(
        run_eval, "run_case",
        lambda case, model, guard: seen.append(graph.BOOKING_ENABLED) or _result(model, "PASS"),
    )
    assert main([*_main_args(golden, cv, out), "--model", "m", "--booking"]) == 0
    assert seen and all(seen)


def test_main_results_files_do_not_collide(run_files, monkeypatch):
    golden, cv, out = run_files
    monkeypatch.setattr(run_eval, "run_case", lambda case, model, guard: _result(model, "PASS"))
    main([*_main_args(golden, cv, out), "--model", "m"])
    main([*_main_args(golden, cv, out), "--model", "m"])
    assert len(list(out.iterdir())) == 2


def test_main_rejects_contact_values_in_history(tmp_path, monkeypatch):
    golden = _write(
        tmp_path,
        [{"id": "h", "question": "q", "history": [{"role": "user", "content": f"I wrote to {EMAIL}"}]}],
    )
    cv = tmp_path / "cv.json"
    cv.write_text(json.dumps(CV))
    monkeypatch.setattr(run_eval, "run_case", lambda *a: pytest.fail("ran a case"))
    assert main([*_main_args(golden, cv, tmp_path / "results"), "--model", "m"]) == 2


@pytest.mark.parametrize("cv_data,expected", [(CV, "contact check: email on, phone on"),
                                              ({}, "contact check: email off, phone off")])
def test_main_reports_contact_guard_state(run_files, monkeypatch, capsys, cv_data, expected):
    golden, cv, out = run_files
    cv.write_text(json.dumps(cv_data))
    monkeypatch.setattr(run_eval, "run_case", lambda case, model, guard: _result(model, "PASS"))
    main([*_main_args(golden, cv, out), "--model", "m"])
    assert expected in capsys.readouterr().out


def test_main_booking_flag_sets_the_prompt_hint(run_files, monkeypatch):
    golden, cv, out = run_files
    monkeypatch.setattr(graph, "BOOKING_ENABLED", False)
    monkeypatch.setattr(llm_messages, "BOOKING_ENABLED", False)
    monkeypatch.setattr(run_eval, "run_case", lambda case, model, guard: _result(model, "PASS"))
    main([*_main_args(golden, cv, out), "--model", "m", "--booking"])
    assert llm_messages.BOOKING_ENABLED is True
