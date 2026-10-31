"""Answer check: ask the bot's Q&A path a golden set of questions and grade the answers.

Each case lists substrings the answer must contain and must not contain. The owner's
email and phone (read from the knowledge JSON at runtime) are always forbidden, and are
never printed or saved: replies are redacted before output.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest.scrub import contact_values, normalize, owner_email_pattern, scrub_text

_ROLES = {"user", "assistant"}
_PHONE_TAIL = 9  # digits compared, so a missing country code still counts as a leak


class GoldenError(ValueError):
    """The golden file is unreadable or a case is malformed."""


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    must_include: list[str | list[str]]
    must_not_include: list[str]
    language: str = "en"
    category: str = ""
    history: list[dict] = field(default_factory=list)


def _is_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parse_case(raw, index: int, seen: set[str]) -> Case:
    if not isinstance(raw, dict):
        raise GoldenError(f"case #{index}: expected a mapping")
    case_id = raw.get("id")
    if not _is_text(case_id):
        raise GoldenError(f"case #{index}: missing or blank id")
    where = f"case {case_id!r}"
    if case_id in seen:
        raise GoldenError(f"{where}: duplicate id")
    if not _is_text(raw.get("question")):
        raise GoldenError(f"{where}: missing or blank question")

    must_include = raw.get("must_include") or []
    if not isinstance(must_include, list):
        raise GoldenError(f"{where}: must_include must be a list")
    for item in must_include:
        if isinstance(item, list):
            if not item or not all(_is_text(alt) for alt in item):
                raise GoldenError(f"{where}: an any-of item must be a non-empty list of strings")
        elif not _is_text(item):
            raise GoldenError(f"{where}: must_include items must be strings or lists of strings")

    must_not_include = raw.get("must_not_include") or []
    if not isinstance(must_not_include, list) or not all(_is_text(s) for s in must_not_include):
        raise GoldenError(f"{where}: must_not_include must be a list of strings")

    history = raw.get("history") or []
    if not isinstance(history, list) or not all(
        isinstance(m, dict) and m.get("role") in _ROLES and isinstance(m.get("content"), str)
        for m in history
    ):
        raise GoldenError(f"{where}: history entries need role user/assistant and string content")

    for key in ("language", "category"):
        if key in raw and not isinstance(raw[key], str):
            raise GoldenError(f"{where}: {key} must be a string")

    return Case(
        id=case_id,
        question=raw["question"],
        must_include=must_include,
        must_not_include=must_not_include,
        language=raw.get("language", "en"),
        category=raw.get("category", ""),
        history=[{"role": m["role"], "content": m["content"]} for m in history],
    )


def load_golden(path: Path) -> list[Case]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise GoldenError(f"cannot read {Path(path).name}: {type(exc).__name__}") from exc
    if data is None or data == []:
        raise GoldenError("golden file is empty")
    if not isinstance(data, list):
        raise GoldenError("golden file must be a list of cases")
    seen: set[str] = set()
    cases = []
    for index, raw in enumerate(data):
        case = _parse_case(raw, index, seen)
        seen.add(case.id)
        cases.append(case)
    return cases


def _fold(text: str) -> str:
    return normalize(text).casefold()


@dataclass(frozen=True)
class ContactGuard:
    email: str | None
    phone_digits: str | None

    @classmethod
    def from_cv(cls, cv: dict) -> ContactGuard:
        email = phone = None
        for value in contact_values(cv):
            if "@" in value:
                email = value.strip()
            else:
                digits = re.sub(r"\D", "", value)
                phone = digits if len(digits) >= _PHONE_TAIL else None
        return cls(email=email, phone_digits=phone)

    def leaks(self, reply: str) -> list[str]:
        """Labels of the owner's contact values found in the reply; never the values."""
        found = []
        text = normalize(reply)
        pattern = owner_email_pattern(self.email) if self.email else None
        if pattern is not None and pattern.search(text):
            found.append("owner email")
        if self.phone_digits and self.phone_digits[-_PHONE_TAIL:] in re.sub(r"\D", "", text):
            found.append("owner phone")
        return found

    def redact(self, reply: str) -> str:
        exact = [v for v in (self.email, self.phone_digits) if v]
        return scrub_text(reply, exact=exact)[0]


@dataclass(frozen=True)
class Grade:
    passed: bool
    missing: list[str]
    forbidden: list[str]


def grade(case: Case, reply: str, guard: ContactGuard) -> Grade:
    if not reply.strip():
        return Grade(passed=False, missing=["(empty reply)"], forbidden=guard.leaks(reply))
    text = _fold(reply)
    missing = []
    for item in case.must_include:
        options = item if isinstance(item, list) else [item]
        if not any(_fold(option) in text for option in options):
            missing.append(" | ".join(options))
    forbidden = [s for s in case.must_not_include if _fold(s) in text]
    forbidden += guard.leaks(reply)
    return Grade(passed=not missing and not forbidden, missing=missing, forbidden=forbidden)
