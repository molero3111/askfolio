"""Answer check: ask the bot's Q&A path a golden set of questions and grade the answers.

Each case lists substrings the answer must contain and must not contain. The owner's
email and phone (read from the knowledge JSON at runtime) are always forbidden, and are
never printed or saved: replies are redacted before output.
"""

import argparse
import json
import logging
import re
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, graph
from app.conversation import ConversationState
from app.llm_client import LLMResult, chat
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


CRITICAL_CATEGORIES = frozenset({"contact", "injection"})
Status = Literal["PASS", "FAIL", "MISROUTED", "ERROR"]


@dataclass
class CaseResult:
    case_id: str
    category: str
    model: str
    served_model: str | None
    status: Status
    missing: list[str]
    forbidden: list[str]
    reply: str
    error: str | None
    latency_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cost: float | None


def run_case(
    case: Case,
    model: str,
    guard: ContactGuard,
    *,
    answer: Callable[..., str] | None = None,
    chat_fn: Callable[..., LLMResult] = chat,
) -> CaseResult:
    """Ask one case through the bot's Q&A path with `model`; never raises."""
    result = CaseResult(
        case_id=case.id, category=case.category, model=model, served_model=None,
        status="ERROR", missing=[], forbidden=[], reply="", error=None, latency_s=None,
        prompt_tokens=None, completion_tokens=None, cost=None,
    )
    if graph._routes_to_scheduling(ConversationState(), case.question):
        result.status = "MISROUTED"
        return result

    usage: list[LLMResult] = []

    def call_llm(messages: list[dict]) -> str:
        llm_result = chat_fn(messages, model=model)
        usage.append(llm_result)
        return llm_result.content

    answer = answer or graph.answer_question
    started = time.monotonic()
    try:
        reply = answer(case.question, case.history, call_llm=call_llm)
    except Exception as exc:
        result.error = guard.redact(f"{type(exc).__name__}: {exc}")
        return result
    finally:
        result.latency_s = round(time.monotonic() - started, 3)
        if usage:
            last = usage[-1]
            result.served_model = last.model
            result.prompt_tokens = last.prompt_tokens
            result.completion_tokens = last.completion_tokens
            result.cost = last.cost

    graded = grade(case, reply, guard)
    result.status = "PASS" if graded.passed else "FAIL"
    result.missing = graded.missing
    result.forbidden = graded.forbidden
    result.reply = guard.redact(reply)
    return result


def summarize(results: list[CaseResult]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for model in dict.fromkeys(r.model for r in results):
        rows = [r for r in results if r.model == model]
        critical = [r for r in rows if r.category in CRITICAL_CATEGORIES]
        latencies = [r.latency_s for r in rows if r.latency_s is not None]
        costs = [r.cost for r in rows if r.cost is not None]
        summary[model] = {
            "passed": sum(r.status == "PASS" for r in rows),
            "total": len(rows),
            "critical_passed": sum(r.status == "PASS" for r in critical),
            "critical_total": len(critical),
            "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "cost": round(sum(costs), 6) if costs else None,
        }
    return summary


def _row(r: CaseResult) -> str:
    latency = f"{r.latency_s:.1f}s" if r.latency_s is not None else "-"
    served = f" (served by {r.served_model})" if r.served_model and r.served_model != r.model else ""
    details = []
    if r.missing:
        details.append("missing: " + "; ".join(r.missing))
    if r.forbidden:
        details.append("forbidden: " + "; ".join(r.forbidden))
    if r.error:
        details.append(r.error)
    return f"{r.status:<9} {r.model}{served}  {r.case_id}  {latency}  {'  '.join(details)}".rstrip()


def _summary_line(model: str, s: dict) -> str:
    latency = f"{s['mean_latency_s']:.1f}s" if s["mean_latency_s"] is not None else "-"
    cost = f"${s['cost']:.4f}" if s["cost"] is not None else "n/a"
    return (
        f"{model}: {s['passed']}/{s['total']} passed, "
        f"critical {s['critical_passed']}/{s['critical_total']}, "
        f"mean latency {latency}, cost {cost}"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--golden", required=True, type=Path, help="golden YAML file")
    parser.add_argument(
        "--model", action="append", default=[],
        help="model to evaluate (repeatable); fallbacks are off when given",
    )
    parser.add_argument("--case", action="append", default=[], help="only run this case id (repeatable)")
    parser.add_argument("--cv", type=Path, default=None, help="knowledge JSON with the owner's contact values")
    parser.add_argument("--out", type=Path, default=ROOT / "evals" / "results", help="results directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    try:
        cases = load_golden(args.golden)
    except GoldenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.case:
        unknown = sorted(set(args.case) - {c.id for c in cases})
        if unknown:
            print(f"error: unknown case id(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c.id in args.case]

    cv_path = args.cv or config.resolve_path(config.INGEST_KNOWLEDGE_JSON)
    try:
        cv = json.loads(Path(cv_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"error: cannot read the knowledge JSON: {type(exc).__name__}", file=sys.stderr)
        return 2
    if not isinstance(cv, dict):
        print("error: the knowledge JSON must be an object", file=sys.stderr)
        return 2
    guard = ContactGuard.from_cv(cv)

    if args.model:
        models = list(dict.fromkeys(args.model))
        config.LLM_FALLBACK_MODELS = []  # measure each model alone
    else:
        models = [config.LLM_MODEL]

    started_at = datetime.now(UTC)
    results = []
    for model in models:
        for case in cases:
            result = run_case(case, model, guard)
            print(_row(result), flush=True)
            results.append(result)

    summary = summarize(results)
    print()
    for model, s in summary.items():
        print(_summary_line(model, s))

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"{started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_file.write_text(
        json.dumps(
            {
                "started_at": started_at.isoformat(),
                "golden": args.golden.name,
                "models": models,
                "results": [asdict(r) for r in results],
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nresults: {out_file}")
    return 0 if all(r.status == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
