"""Fail CI when a tracked file contains a term from the PRIVACY_DENYLIST secret.

Matched terms are never printed; only ``path:line`` locations are reported.
"""

import os
import re
import subprocess
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path

_MIN_DIGITS = 7
_DIGIT_RATIO = 0.7
_NATIONAL_DIGITS = 10
_BINARY_SNIFF_BYTES = 8192


def _norm(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _digit_needles(term: str) -> list[str]:
    """Digit-only needles for phone-like terms (mostly digits, at least 7 of them)."""
    compact = "".join(term.split())
    digits = "".join(c for c in compact if c.isdigit())
    if len(digits) < _MIN_DIGITS or len(digits) / len(compact) < _DIGIT_RATIO:
        return []
    needles = [digits]
    if len(digits) > _NATIONAL_DIGITS:
        # also the number without its country code
        needles.append(digits[-_NATIONAL_DIGITS:])
    return needles


def scan(files: Iterable[Path], terms: list[str]) -> list[tuple[str, int]]:
    """Return (path, line_no) for every line containing any term.

    Comparison is NFKC + casefold on both sides. Phone-like terms also match on the
    line's digits alone, so formatting differences do not hide them.
    """
    needles = [_norm(t) for t in terms if t.strip()]
    digit_needles = [d for t in terms if t.strip() for d in _digit_needles(_norm(t))]
    hits: list[tuple[str, int]] = []
    if not needles:
        return hits
    for path in files:
        try:
            raw = Path(path).read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:_BINARY_SNIFF_BYTES]:
            continue
        text = raw.decode("utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            normalized = _norm(line)
            if any(n in normalized for n in needles):
                hits.append((str(path), line_no))
                continue
            if digit_needles:
                line_digits = "".join(c for c in normalized if c.isdigit())
                if any(d in line_digits for d in digit_needles):
                    hits.append((str(path), line_no))
    return hits


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], check=True, capture_output=True, text=True
    ).stdout
    return [Path(p) for p in out.split("\0") if p]


def main() -> int:
    raw = os.environ.get("PRIVACY_DENYLIST", "")
    terms = [t.strip() for t in re.split(r"[,\r\n]+", raw)]
    terms = [t for t in terms if t]
    if raw and not terms:
        print("::error::PRIVACY_DENYLIST is set but contains no usable terms")
        return 2
    if not terms and os.environ.get("REQUIRE_DENYLIST", "").strip().lower() in {"true", "1"}:
        print("::error::PRIVACY_DENYLIST is required here but is empty or unset")
        return 2
    if not terms:
        print("::notice::PRIVACY_DENYLIST not set; privacy scan skipped")
        return 0
    hits = scan(tracked_files(), terms)
    for path, line_no in hits:
        print(f"{path}:{line_no}")
    if hits:
        print(f"Privacy scan failed: {len(hits)} hit(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
