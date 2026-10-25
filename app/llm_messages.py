import re

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


def build_qa_messages(context: str, chat_history: str, user_input: str) -> list[dict]:
    system = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    user = USER_PROMPT_PATH.read_text(encoding="utf-8").format(
        context=neutralize_tags(context),
        chat_history=neutralize_tags(chat_history),
        user_input=neutralize_tags(user_input),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_timezone_messages(user_input: str) -> list[dict]:
    system = TIMEZONE_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    user = TIMEZONE_PROMPT_PATH.read_text(encoding="utf-8").format(user_input=user_input)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
