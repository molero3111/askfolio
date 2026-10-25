import re

from app.config import (
    BOOKING_ENABLED,
    SYSTEM_PROMPT_PATH,
    TIMEZONE_PROMPT_PATH,
    TIMEZONE_SYSTEM_PROMPT_PATH,
    USER_PROMPT_PATH,
)
from app.text_normalize import normalize_text

_TAG_RE = re.compile(
    r"<\s*(/?)\s*(profile_context|chat_history|question)\b[^>]*>",
    re.IGNORECASE,
)


def neutralize_tags(text: str) -> str:
    """Replace our delimiter tags with bracketed look-alikes so input cannot close a block."""
    return _TAG_RE.sub(lambda m: f"[{m.group(1)}{m.group(2)}]", text)


def sanitize(text: str) -> str:
    """Fold look-alike/invisible/encoded forms, neutralise our tags, escape any other angle bracket."""
    text = neutralize_tags(normalize_text(text))
    return text.replace("<", "\u2039").replace(">", "\u203a")


_HINT_BOOKING_ON = "the profile links (such as LinkedIn and GitHub) and offer to schedule a meeting."
_HINT_BOOKING_OFF = (
    "the profile links (LinkedIn, GitHub, portfolio). "
    "Meeting scheduling isn't available through this assistant."
)


_RULE_BOOKING_ON = (
    "If someone wants to schedule a meeting, call or interview, say you can help arrange one."
)
_RULE_BOOKING_OFF = (
    "If someone asks to schedule a meeting, call or interview, explain that scheduling isn't "
    "available through this assistant and point them to the profile links (LinkedIn, GitHub, portfolio)."
)


def build_qa_messages(context: str, chat_history: str, user_input: str) -> list[dict]:
    hint = _HINT_BOOKING_ON if BOOKING_ENABLED else _HINT_BOOKING_OFF
    rule = _RULE_BOOKING_ON if BOOKING_ENABLED else _RULE_BOOKING_OFF
    # replace(), not format(): the system prompt is free text that may contain braces.
    system = (
        SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        .strip()
        .replace("{scheduling_hint}", hint)
        .replace("{scheduling_rule}", rule)
    )
    user = USER_PROMPT_PATH.read_text(encoding="utf-8").format(
        context=sanitize(context),
        chat_history=sanitize(chat_history),
        user_input=sanitize(user_input),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_timezone_messages(user_input: str) -> list[dict]:
    system = TIMEZONE_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    user = TIMEZONE_PROMPT_PATH.read_text(encoding="utf-8").format(user_input=sanitize(user_input))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
