"""Shared text normaliser: decode entities, fold look-alikes, drop invisible characters."""

import html
import re
import unicodedata

# Zero-width / format characters that split tokens without changing how they read.
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff\u00ad]")
_MAX_UNESCAPE_PASSES = 5


def normalize_text(s: str) -> str:
    """Unescape entities, apply NFKC and strip invisible characters until stable (max 5 passes).

    The whole step repeats because folding or stripping can itself form a new entity,
    e.g. a fullwidth ``＆lt;`` or ``&l<ZWSP>t;`` only becomes ``&lt;`` after the first pass.
    """
    for _ in range(_MAX_UNESCAPE_PASSES):
        step = _INVISIBLE_RE.sub("", unicodedata.normalize("NFKC", html.unescape(s)))
        if step == s:
            break
        s = step
    return s
