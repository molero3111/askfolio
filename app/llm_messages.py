import html
import re
import unicodedata

from app.config import (
    SYSTEM_PROMPT_PATH,
    TIMEZONE_PROMPT_PATH,
    TIMEZONE_SYSTEM_PROMPT_PATH,
    USER_PROMPT_PATH,
)

_TAG_RE = re.compile(
    r"<\s*(/?)\s*(profile_context|chat_history|question)\b[^>]*>",
    re.IGNORECASE,
)


def neutralize_tags(text: str) -> str:
    """Replace our delimiter tags with bracketed look-alikes so input cannot close a block."""
    return _TAG_RE.sub(lambda m: f"[{m.group(1)}{m.group(2)}]", text)


_INVISIBLE_RE = re.compile("[\u200b-\u200f\u2060\ufeff\u00ad]")


def sanitize(text: str) -> str:
    """Fold look-alike/invisible/encoded forms, neutralise our tags, escape any other angle bracket."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = html.unescape(text)
    text = neutralize_tags(text)
    return text.replace("<", "\u2039").replace(">", "\u203a")


def build_qa_messages(context: str, chat_history: str, user_input: str) -> list[dict]:
    system = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
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
