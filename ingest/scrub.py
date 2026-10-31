"""Remove email addresses and phone numbers from documents before embedding."""

import re
from collections.abc import Sequence
from urllib.parse import unquote

from langchain_core.documents import Document

from app.text_normalize import normalize_text

REPLACEMENT = "[contact removed]"

# Not preceded by local-part characters (avoids quadratic rescans) nor by "//" (URL
# userinfo); "git@host:path" scp-style remotes are not emails.
_EMAIL_RE = re.compile(
    r"(?:mailto:)?(?<![A-Za-z0-9._%+-])(?<!//)(?!git@[^\s:]+:)"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
_PHONE_RE = re.compile(r"(?<![\w/=.:-])\+?\(?\d[\d \t().-]{7,}\d(?![\w/-])")
_YEAR_RE = re.compile(r"(?:19|20)\d\d")


_PCT_AT_TOKEN_RE = re.compile(r"\S*%40\S*", re.IGNORECASE)
_AT_SEP = r"\s*(?:@|\[at\]|\(at\)|\[@\]|\(@\)|\bat\b|\barroba\b)\s*"
_DOT_SEP = r"(?:\s*(?:\.|\[dot\]|\(dot\)|\bdot\b|\bpunto\b)\s*)"


def normalize(text: str) -> str:
    """Shared normalisation plus the scrub-specific '%40' unquote of percent-encoded emails."""
    text = normalize_text(text)
    if "%40" in text.lower():
        text = _PCT_AT_TOKEN_RE.sub(lambda m: unquote(m.group(0)), text)
        text = normalize_text(text)
    return text


def owner_email_pattern(value: str) -> re.Pattern[str] | None:
    """Match the owner's email incl. '[at]' / ' at ' / ' dot ' spellings, case-insensitive."""
    local, sep, domain = normalize(value).strip().partition("@")
    if not (sep and local and domain):
        return None
    local_re = _DOT_SEP.join(re.escape(part) for part in local.split("."))
    domain_re = _DOT_SEP.join(re.escape(part) for part in domain.split("."))
    return re.compile(
        r"(?:mailto:)?(?<![A-Za-z0-9._%+-])" + local_re + _AT_SEP + domain_re,
        re.IGNORECASE,
    )


def _looks_like_phone(candidate: str) -> bool:
    runs = re.findall(r"\d+", candidate)
    if not 9 <= sum(len(r) for r in runs) <= 15:
        return False
    # Dotted versions and IPs ("3.10 3.11", "192.168.100.200"): every run is short.
    if "." in candidate and all(len(r) <= 3 for r in runs):
        return False
    # Year lists and ranges ("2018-2020 (2 years)").
    is_year_list = all(len(r) <= 4 for r in runs) and sum(bool(_YEAR_RE.fullmatch(r)) for r in runs) >= 2
    return not is_year_list


def contact_values(cv: dict) -> list[str]:
    info = cv.get("personal_info") or {}
    return [v for v in (info.get("email"), info.get("phone")) if isinstance(v, str) and v.strip()]


def _exact_pattern(value: str) -> re.Pattern[str] | None:
    value = value.strip()
    if "@" in value:
        return owner_email_pattern(value)
    digits = [c for c in value if c.isdigit()]
    if not digits:
        return None
    body = r"[\s().-]*".join(digits)
    return re.compile(r"(?<!\d)\+?\(?" + body + r"(?!\d)")


def _scrub(text: str, exact: Sequence[str]) -> tuple[str, list[tuple[str, int]]]:
    """Return scrubbed text and (kind, matched length) per removal."""
    found: list[tuple[str, int]] = []
    text = normalize(text)

    for value in exact:
        pattern = _exact_pattern(value)
        if pattern is None:
            continue
        text = pattern.sub(lambda m: found.append(("exact", len(m.group(0)))) or REPLACEMENT, text)

    if "@" in text:
        text = _EMAIL_RE.sub(lambda m: found.append(("email", len(m.group(0)))) or REPLACEMENT, text)

    def _phone(m: re.Match[str]) -> str:
        if _looks_like_phone(m.group(0)):
            found.append(("phone", len(m.group(0))))
            return REPLACEMENT
        return m.group(0)

    text = _PHONE_RE.sub(_phone, text)
    return text, found


def scrub_text(text: str, exact: Sequence[str] = ()) -> tuple[str, list[str]]:
    text, found = _scrub(text, exact)
    return text, [kind for kind, _ in found]


def scrub_documents(docs: list[Document], exact: Sequence[str]) -> list[Document]:
    out: list[Document] = []
    for doc in docs:
        found: list[tuple[str, int]] = []
        metadata = {}
        for k, v in doc.metadata.items():
            if isinstance(v, str):
                v, f = _scrub(v, exact)
                found.extend(f)
            metadata[k] = v
        content, f = _scrub(doc.page_content, exact)
        found.extend(f)
        where = f"{metadata.get('source', '?')}/{metadata.get('section', '?')}"
        for kind, length in found:
            print(f"[scrub] removed {kind} from {where} ({length} chars)")
        out.append(Document(page_content=content, metadata=metadata))
    return out
