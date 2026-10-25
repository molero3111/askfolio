"""Shared text normaliser: decode entities, fold look-alikes, drop invisible characters."""

import html
import re
import unicodedata

# Zero-width / format characters that split tokens without changing how they read.
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff\u00ad]")
_MAX_UNESCAPE_PASSES = 5


def normalize_text(s: str) -> str:
    """Unescape HTML entities until stable (max 5 passes), apply NFKC, strip invisible characters."""
    for _ in range(_MAX_UNESCAPE_PASSES):
        unescaped = html.unescape(s)
        if unescaped == s:
            break
        s = unescaped
    s = unicodedata.normalize("NFKC", s)
    return _INVISIBLE_RE.sub("", s)
