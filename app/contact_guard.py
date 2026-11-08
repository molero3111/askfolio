"""Spots the owner's email and phone in text the LLM wrote."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import cache
from unicodedata import digit

from app import config
from ingest.scrub import contact_values, normalize, scrub_text

logger = logging.getLogger(__name__)

_PHONE_TAIL = 9  # digits compared, so a missing country code still counts as a leak
_PHONE_MIN = 7  # fewer digits would match ordinary numbers in a reply
WITHHELD = "[reply withheld: contact leak]"
_PLUS_TAG = re.compile(r"(?<=\w)\+[^\s@]*(?=\W*(?:@|at\b|arroba\b))", re.IGNORECASE)


def _compact(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


def _digits(text: str) -> str:
    return "".join(str(digit(c)) for c in text if c.isdigit())


def _compact_email_pattern(email: str) -> re.Pattern[str] | None:
    """Match the email in compacted text, so any separator or spacing a reply uses still counts."""
    local, _, domain = normalize(email).strip().partition("@")
    local_parts = [_compact(part) for part in local.split(".")]
    labels = [_compact(label) for label in domain.split(".")]
    if not all(local_parts) or not all(labels):
        return None
    optional_dot = "(?:dot|punto)?"
    return re.compile(optional_dot.join(local_parts) + "(?:at|arroba)?" + optional_dot.join(labels))


@dataclass(frozen=True)
class ContactGuard:
    email: str | None
    phone_digits: str | None

    @classmethod
    def from_cv(cls, cv: dict) -> ContactGuard:
        email = phone = None
        for value in contact_values(cv):
            if "@" in value:
                email = value.strip() if _compact_email_pattern(value) else None
            else:
                digits = _digits(value)
                phone = digits if len(digits) >= _PHONE_MIN else None
        return cls(email=email, phone_digits=phone)

    def leaks(self, reply: str) -> list[str]:
        """Labels of the owner's contact values found in the reply; never the values."""
        found = []
        text = normalize(reply)
        pattern = _compact_email_pattern(self.email) if self.email else None
        # Both forms: a reply may add a plus tag, and the owner's own address may hold one.
        if pattern is not None and (
            pattern.search(_compact(text)) or pattern.search(_compact(_PLUS_TAG.sub("", text)))
        ):
            found.append("owner email")
        # last 9 digits, so a dropped country code still counts; a shorter phone in full
        if self.phone_digits and self.phone_digits[-_PHONE_TAIL:] in _digits(text):
            found.append("owner phone")
        return found

    def redact(self, reply: str) -> str:
        exact = [v for v in (self.email, self.phone_digits) if v]
        redacted = scrub_text(reply, exact=exact)[0]
        # The scrubber misses some separators the digit check catches; never show those.
        return WITHHELD if self.leaks(redacted) else redacted


@cache
def get_guard() -> ContactGuard | None:
    """The guard for the configured knowledge JSON, or None (warned once) if it can't be built."""
    path = config.resolve_path(config.INGEST_KNOWLEDGE_JSON)
    try:
        cv = json.loads(path.read_text(encoding="utf-8"))
        guard = ContactGuard.from_cv(cv)
    except (OSError, ValueError, AttributeError) as exc:
        logger.error("[contact_guard] knowledge JSON unusable (%s); replies go unchecked", type(exc).__name__)
        return None
    if not (guard.email or guard.phone_digits):
        logger.error("[contact_guard] no owner email or phone in the knowledge JSON; replies go unchecked")
        return None
    if not guard.email:
        logger.warning("[contact_guard] owner email not in the knowledge JSON; only the phone is checked")
    if not guard.phone_digits:
        logger.warning("[contact_guard] owner phone not in the knowledge JSON; only the email is checked")
    labels = [label for label, value in (("owner email", guard.email), ("owner phone", guard.phone_digits)) if value]
    logger.info("[contact_guard] checking %s", ", ".join(labels))
    return guard
