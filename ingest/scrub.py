"""Remove email addresses and phone numbers from documents before embedding."""

import re
from collections.abc import Sequence

from langchain_core.documents import Document

REPLACEMENT = "[contact removed]"

_EMAIL_RE = re.compile(r"(?:mailto:)?[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<![\w/=.:-])\+?\(?\d[\d\s().-]{7,}\d(?![\w/-])")


def contact_values(cv: dict) -> list[str]:
    info = cv.get("personal_info") or {}
    return [v for v in (info.get("email"), info.get("phone")) if isinstance(v, str) and v.strip()]


def _exact_pattern(value: str) -> re.Pattern[str] | None:
    value = value.strip()
    if "@" in value:
        return re.compile(r"(?:mailto:)?" + re.escape(value), re.IGNORECASE)
    digits = [c for c in value if c.isdigit()]
    if not digits:
        return None
    body = r"[\s().-]*".join(digits)
    return re.compile(r"(?<!\d)\+?\(?" + body + r"(?!\d)")


def _scrub(text: str, exact: Sequence[str]) -> tuple[str, list[tuple[str, int]]]:
    """Return scrubbed text and (kind, matched length) per removal."""
    found: list[tuple[str, int]] = []

    for value in exact:
        pattern = _exact_pattern(value)
        if pattern is None:
            continue
        text = pattern.sub(lambda m: found.append(("exact", len(m.group(0)))) or REPLACEMENT, text)

    text = _EMAIL_RE.sub(lambda m: found.append(("email", len(m.group(0)))) or REPLACEMENT, text)

    def _phone(m: re.Match[str]) -> str:
        if 9 <= sum(c.isdigit() for c in m.group(0)) <= 15:
            found.append(("phone", len(m.group(0))))
            return REPLACEMENT
        return m.group(0)

    text = _PHONE_RE.sub(_phone, text)
    return text, found


def scrub_text(text: str, exact: Sequence[str] = ()) -> tuple[str, list[str]]:
    text, found = _scrub(text, exact)
    return text, [kind for kind, _ in found]


def _clean(text: str, exact: Sequence[str], where: str) -> str:
    new, found = _scrub(text, exact)
    for kind, length in found:
        print(f"[scrub] removed {kind} from {where} ({length} chars)")
    return new


def scrub_documents(docs: list[Document], exact: Sequence[str]) -> list[Document]:
    out: list[Document] = []
    for doc in docs:
        where = f"{doc.metadata.get('source', '?')}/{doc.metadata.get('section', '?')}"
        metadata = {
            k: _clean(v, exact, where) if isinstance(v, str) else v for k, v in doc.metadata.items()
        }
        out.append(Document(page_content=_clean(doc.page_content, exact, where), metadata=metadata))
    return out
