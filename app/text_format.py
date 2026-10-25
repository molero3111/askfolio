"""Backstop that turns stray Markdown into plain text before a reply is sent to Telegram.

Telegram messages are sent without a parse mode, so Markdown would show up literally. The prompt
asks the model for plain text; this keeps that promise when the model ignores it.
"""

import re

_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_BOLD_RE = re.compile(r"\*\*([^\n]+?)\*\*")
_CODE_RE = re.compile(r"`([^`\n]+)`")
_STAR_BULLET_RE = re.compile(r"^(\s*)\* ", re.MULTILINE)
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)


def _link(match: re.Match) -> str:
    label, url = match.group(1).strip(), match.group(2)
    return url if label == url else f"{label}: {url}"


def to_plain_text(text: str) -> str:
    text = _LINK_RE.sub(_link, text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _CODE_RE.sub(r"\1", text)
    text = _STAR_BULLET_RE.sub(r"\1• ", text)
    return _HEADING_RE.sub("", text)
