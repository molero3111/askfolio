from pathlib import Path

import pytest
import yaml

from scripts.run_eval import Case, ContactGuard, GoldenError, grade, load_golden

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


def test_redact_hides_values(guard):
    out = guard.redact(f"Mail {EMAIL} or call {PHONE}")
    assert "jane.doe" not in out
    assert "5550100" not in out.replace(" ", "")


def test_guard_without_contacts():
    empty = ContactGuard.from_cv({})
    assert empty.leaks(f"{EMAIL} {PHONE}") == []
    assert empty.redact("plain") == "plain"


def test_empty_reply_fails(guard):
    result = grade(_case(), "", guard)
    assert not result.passed
    assert result.missing == ["(empty reply)"]
